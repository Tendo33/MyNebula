import { FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { Shield, User } from 'lucide-react';

import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from '@/components/ui/input-group';
import { Spinner } from '@/components/ui/spinner';

interface SettingsLoginFormProps {
  loginUsername: string;
  loginPassword: string;
  loginLoading: boolean;
  loginError: string | null;
  adminAuthConfigured: boolean | null;
  onUsernameChange: (v: string) => void;
  onPasswordChange: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
}

export const SettingsLoginForm = ({
  loginUsername,
  loginPassword,
  loginLoading,
  loginError,
  adminAuthConfigured,
  onUsernameChange,
  onPasswordChange,
  onSubmit,
}: SettingsLoginFormProps) => {
  const { t } = useTranslation();

  return (
    <section className="flex flex-1 items-center justify-center px-6 py-10">
      <Card className="w-full max-w-lg p-7 sm:p-8">
        <CardHeader className="mb-6 flex flex-row items-start gap-4 p-0">
          <div className="flex size-8 items-center justify-center rounded-md border border-border bg-card text-foreground">
            <Shield />
          </div>
          <div>
            <CardTitle className="text-xl font-semibold">{t('settings.admin_access')}</CardTitle>
            <CardDescription className="mt-1">{t('settings.login_required_desc')}</CardDescription>
          </div>
        </CardHeader>

        <CardContent className="p-0">
          <form className="flex flex-col gap-4" onSubmit={onSubmit}>
            {adminAuthConfigured === false && (
              <Alert variant="warning">
                <AlertDescription>{t('settings.admin_not_configured')}</AlertDescription>
              </Alert>
            )}

            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="admin-username">{t('settings.username')}</FieldLabel>
                <InputGroup>
                  <InputGroupAddon>
                    <User />
                  </InputGroupAddon>
                  <InputGroupInput
                    id="admin-username"
                    type="text"
                    value={loginUsername}
                    onChange={(e) => onUsernameChange(e.target.value)}
                    autoComplete="username"
                    required
                  />
                </InputGroup>
              </Field>

              <Field>
                <FieldLabel htmlFor="admin-password">{t('settings.password')}</FieldLabel>
                <Input
                  id="admin-password"
                  type="password"
                  value={loginPassword}
                  onChange={(e) => onPasswordChange(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </Field>
            </FieldGroup>

            {loginError && (
              <Alert variant="destructive">
                <AlertDescription>{loginError}</AlertDescription>
              </Alert>
            )}

            <Button
              type="submit"
              disabled={loginLoading || adminAuthConfigured === false}
              className="w-full"
            >
              {loginLoading ? <Spinner data-icon="inline-start" /> : null}
              {t('app.login')}
            </Button>
          </form>
        </CardContent>
      </Card>
    </section>
  );
};
